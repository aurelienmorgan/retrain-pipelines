"""context store

Revision ID: 450e163fd43d
Revises: bdff337fbb86
Create Date: 2026-07-13 16:12:25.668620

"""
import os
from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

DEFAULT_METADATA_ROOT = os.path.join(os.environ["RP_ASSETS_CACHE"], "metadata")

# revision identifiers, used by Alembic.
revision: str = '450e163fd43d'
down_revision: Union[str, Sequence[str], None] = 'bdff337fbb86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    ######################
    ## create new table ##
    ######################
    op.create_table('task_context_attrs',
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.Column('attr_name', sa.String(), nullable=False),
        sa.Column('sha', sa.String(), nullable=False),
        sa.Column('disk_ref', sa.String(), nullable=True),
        sa.Column('inline_val', sa.JSON(none_as_null=True), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('task_id', 'attr_name')
    )
    ######################

    ###########################################
    ## drop "executions.context_dump" column ##
    ###########################################
    op.drop_column('executions', 'context_dump')
    ###########################################

    ##############################################################
    ## insert "executions.metadata_root" column in 4th position ##
    ##############################################################
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # 1. Get existing columns and primary-keys info
    cols = inspector.get_columns("executions")
    names = [c["name"] for c in cols]
    types = [c["type"] for c in cols]
    nullables = [c.get("nullable", True) for c in cols]

    pk_info = inspector.get_pk_constraint("executions")
    pk_columns = set(pk_info.get("constrained_columns", []))
    pks = [name in pk_columns for name in names]

    # 2. Build new column list with metadata_root at 4th position (index 3)
    new_names = names[:3] + ["metadata_root"] + names[3:]
    new_types = types[:3] + [sa.String()] + types[3:]
    new_nullables = nullables[:3] + [False] + nullables[3:]
    new_pks = pks[:3] + [False] + pks[3:]

    new_cols = [
        sa.Column(n, t, nullable=nul, primary_key=pk)
        for n, t, nul, pk in zip(new_names, new_types, new_nullables, new_pks)
    ]

    # 3. Get foreign keys and indexes before dropping the table
    #
    # Find all foreign keys that reference executions.
    referencing_fks: list[dict[str, Any]] = []
    for table_name in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(table_name):
            if fk.get("referred_table") == "executions":
                referencing_fks.append(
                    {
                        "table": table_name,
                        "name": fk.get("name"),
                        "constrained_columns": fk.get("constrained_columns"),
                        "referred_columns": fk.get("referred_columns"),
                    }
                )

    indexes = inspector.get_indexes("executions")

    # 4. Drop FKs that reference executions
    #
    # Use direct constraint operations when supported.
    # Otherwise dependent objects must be handled by the dialect.
    if conn.dialect.supports_alter:
        for fk_ref in referencing_fks:
            table = fk_ref["table"]
            name = fk_ref["name"]
            if name is None:
                continue
            op.drop_constraint(name, table_name=table, type_="foreignkey")

    # 5. Create new table with reordered columns
    op.create_table("executions_new", *new_cols)

    # 6. Copy data, setting metadata_root for all existing rows
    sel = ", ".join(names)
    ins = sel + ", metadata_root"
    op.execute(
        sa.text(
            f"INSERT INTO executions_new ({ins}) "
            f"SELECT {sel}, :default FROM executions"
        ).bindparams(default=DEFAULT_METADATA_ROOT)
    )

    # 7. Drop old table (now no FKs depend on it)
    op.drop_table("executions")

    # 8. Rename new table
    op.rename_table("executions_new", "executions")

    # 9. Re-apply foreign keys (only the ones that were removed)
    if conn.dialect.supports_alter:
        for fk_ref in referencing_fks:
            table = fk_ref["table"]
            name = fk_ref["name"]
            constrained = fk_ref["constrained_columns"]
            referred = fk_ref["referred_columns"]
            if name is None or not constrained or not referred:
                continue
            op.create_foreign_key(name, table, "executions", constrained, referred)


    # 10. Re-apply indexes (skip primary key index if it exists)
    for idx in indexes:
        idx_cols = idx.get("column_names") or []
        idx_col_names = [c for c in idx_cols if c is not None]
        name = idx.get("name")
        if name is None or not idx_col_names:
            continue
        op.create_index(name, "executions", idx_col_names, unique=idx.get("unique", False))
    ##############################################################

def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('executions', sa.Column('context_dump', sqlite.JSON(), nullable=True))
    op.drop_column('executions', 'metadata_root')
    op.drop_table('task_context_attrs')
    # ### end Alembic commands ###
